import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { User, UserDocument } from './schemas/user.schema';
import * as bcrypt from 'bcryptjs';

@Injectable()
export class UserService {
  constructor(@InjectModel(User.name) private userModel: Model<UserDocument>) {}

  // Create a new user (signup)
  async createUser(
    username: string,
    email: string,
    password: string,
  ): Promise<User> {
    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new this.userModel({
      username,
      email,
      password: hashedPassword,
    });
    return newUser.save();
  }

  // Find user by username
  async findByUsername(username: string): Promise<User | undefined> {
    return this.userModel.findOne({ username }).exec();
  }

  // Validate a user with username and password (login)
  async validateUser(username: string, password: string): Promise<User | null> {
    const user = await this.findByUsername(username);
    if (user && (await bcrypt.compare(password, user.password))) {
      return user;
    }
    return null;
  }
}
