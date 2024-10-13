import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { AuthController } from './auth/auth.controller';
import { AuthService } from './auth/auth.service';
import { JwtModule } from '@nestjs/jwt';
import { PassportModule } from '@nestjs/passport';
import { JwtStrategy } from './auth/jwt.strategy';
import { UserService } from './user/user.service';
import { User, UserSchema } from './user/schemas/user.schema';

@Module({
  imports: [
    MongooseModule.forRoot('mongodb://localhost:27017/ps2speedrun'), // MongoDB connection
    MongooseModule.forFeature([{ name: User.name, schema: UserSchema }]), // User schema
    PassportModule,
    JwtModule.register({
      secret: 'secretKey', // Use env variables for this in production
      signOptions: { expiresIn: '1h' }, // JWT expiration
    }),
  ],
  controllers: [AuthController], // Register controllers
  providers: [
    AuthService,
    UserService,
    JwtStrategy,
  ], // Register services
})
export class AppModule {}
