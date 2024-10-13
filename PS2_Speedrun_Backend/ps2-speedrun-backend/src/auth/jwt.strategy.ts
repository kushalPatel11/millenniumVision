import { Injectable } from '@nestjs/common';
import { PassportStrategy } from '@nestjs/passport';
import { ExtractJwt, Strategy } from 'passport-jwt';
import { UserService } from '../user/user.service';
import { JwtPayload } from './interfaces/jwt-payload.interface';
import { User } from '../user/schemas/user.schema';

@Injectable()
export class JwtStrategy extends PassportStrategy(Strategy) {
  constructor(private userService: UserService) {
    super({
      jwtFromRequest: ExtractJwt.fromAuthHeaderAsBearerToken(),
      secretOrKey: 'secretKey', // Use an env variable in production
    });
  }

  async validate(payload: JwtPayload): Promise<User> {
    return this.userService.findByUsername(payload.username);
  }
}
