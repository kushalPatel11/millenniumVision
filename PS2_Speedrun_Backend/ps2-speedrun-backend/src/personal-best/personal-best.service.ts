import { Injectable } from '@nestjs/common';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import {
  PersonalBest,
  PersonalBestDocument,
} from './schema/personal-best.schema';
import { CreatePersonalBestDto } from './dto/create-personal-best.dto';

@Injectable()
export class PersonalBestService {
  constructor(
    @InjectModel(PersonalBest.name)
    private personalBestModel: Model<PersonalBestDocument>,
  ) {}

  async findByUserId(userId: string): Promise<PersonalBest> {
    return this.personalBestModel.findOne({ userId }).exec();
  }

  async createOrUpdate(
    userId: string,
    personalBest: number,
  ): Promise<PersonalBest> {
    return this.personalBestModel
      .findOneAndUpdate(
        { userId },
        { userId, personalBest },
        { new: true, upsert: true }, // Create or update personal best
      )
      .exec();
  }
}
